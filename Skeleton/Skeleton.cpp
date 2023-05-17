//=============================================================================================
// Mintaprogram: Zöld háromszög. Ervenyes 2019. osztol.
//
// A beadott program csak ebben a fajlban lehet, a fajl 1 byte-os ASCII karaktereket tartalmazhat, BOM kihuzando.
// Tilos:
// - mast "beincludolni", illetve mas konyvtarat hasznalni
// - faljmuveleteket vegezni a printf-et kiveve
// - Mashonnan atvett programresszleteket forrasmegjeloles nelkul felhasznalni es
// - felesleges programsorokat a beadott programban hagyni!!!!!!! 
// - felesleges kommenteket a beadott programba irni a forrasmegjelolest kommentjeit kiveve
// ---------------------------------------------------------------------------------------------
// A feladatot ANSI C++ nyelvu forditoprogrammal ellenorizzuk, a Visual Studio-hoz kepesti elteresekrol
// es a leggyakoribb hibakrol (pl. ideiglenes objektumot nem lehet referencia tipusnak ertekul adni)
// a hazibeado portal ad egy osszefoglalot.
// ---------------------------------------------------------------------------------------------
// A feladatmegoldasokban csak olyan OpenGL fuggvenyek hasznalhatok, amelyek az oran a feladatkiadasig elhangzottak 
// A keretben nem szereplo GLUT fuggvenyek tiltottak.
//
// NYILATKOZAT
// ---------------------------------------------------------------------------------------------
// Nev    : Ursuleac Zsolt
// Neptun : S8H56Y
// ---------------------------------------------------------------------------------------------
// ezennel kijelentem, hogy a feladatot magam keszitettem, es ha barmilyen segitseget igenybe vettem vagy
// mas szellemi termeket felhasznaltam, akkor a forrast es az atvett reszt kommentekben egyertelmuen jeloltem.
// A forrasmegjeloles kotelme vonatkozik az eloadas foliakat es a targy oktatoi, illetve a
// grafhazi doktor tanacsait kiveve barmilyen csatornan (szoban, irasban, Interneten, stb.) erkezo minden egyeb
// informaciora (keplet, program, algoritmus, stb.). Kijelentem, hogy a forrasmegjelolessel atvett reszeket is ertem,
// azok helyessegere matematikai bizonyitast tudok adni. Tisztaban vagyok azzal, hogy az atvett reszek nem szamitanak
// a sajat kontribucioba, igy a feladat elfogadasarol a tobbi resz mennyisege es minosege alapjan szuletik dontes.
// Tudomasul veszem, hogy a forrasmegjeloles kotelmenek megsertese eseten a hazifeladatra adhato pontokat
// negativ elojellel szamoljak el es ezzel parhuzamosan eljaras is indul velem szemben.
//=============================================================================================
#include "framework.h"

bool nowPrint = false;

struct Material {
	vec3 ka, kd, ks;
	float  shininess;
	Material(vec3 _kd, vec3 _ks, float _shininess) : ka(_kd* M_PI), kd(_kd), ks(_ks) { shininess = _shininess; }
};

struct Hit {
	float t;
	vec3 position, normal;
	Material* material;
	Hit() { t = -1; }
};

struct Ray {
	vec3 start, dir;
	Ray(vec3 _start, vec3 _dir) { start = _start; dir = normalize(_dir); }
};

class Intersectable {
protected:
	Material* material;
public:
	virtual Hit intersect(const Ray& ray) = 0;
};

struct Triangle : Intersectable {
	vec3 p1, p2, p3;

	Triangle(vec3 pi1, vec3 pi2, vec3 pi3) {
		p1 = pi1;
		p2 = pi2;
		p3 = pi3;
	}

	Hit intersect(const Ray& ray) {
		Hit inters;
		vec3 n = cross(p2 - p1, p3 - p1);
		float t = dot((p1 - ray.start), n) / dot(ray.dir, n);

		vec3 p = ray.start + ray.dir * t;

		if (!	(dot(cross(p2 - p1, p - p1), n) > 0 &&
				dot(cross(p3 - p2, p - p2), n) > 0 &&
				dot(cross(p1 - p3, p - p3), n) > 0))
					return inters;

		inters.normal = -normalize(n);
		inters.position = p;
		inters.t = t;
		
		return inters;

	}
};
struct Box : Intersectable {
	struct Face : Intersectable {
		vec3 p1, p2, p3, p4;
		vec3 n;

		Face(vec3 x1, vec3 x2, vec3 x3, vec3 x4, vec3 normal) {
			p1 = x1; p2 = x2; p3 = x3; p4 = x4;
			n = normal;
		}

		Hit intersect(const Ray& ray) {
			Hit hit;

			float t = dot(p1 - ray.start, n) / dot(ray.dir, n);
			if (t < 0) return hit;

			vec3 p = ray.start + ray.dir * t;

			if (!(dot(cross(p2 - p1, p - p1), n) > 0 &&
				dot(cross(p3 - p2, p - p2), n) > 0 &&
				dot(cross(p4 - p3, p - p3), n) > 0 &&
				dot(cross(p1 - p4, p - p4), n) > 0))
				return hit;

			if (dot(ray.dir, n) <= 0)
				return hit;

			hit.t = t;
			hit.normal = (dot(ray.dir,n)>0) ? -n : n;
			hit.position = ray.start + ray.dir * t;

			return hit;
		}

	};

	std::vector<Intersectable*> faces;

	Box() {
		faces.push_back( new Face(vec3(1, 0, 1), vec3(0, 0, 1), vec3(0, 0, 0), vec3(1, 0, 0), vec3(0, -1, 0)));
		faces.push_back(new Face(vec3(0, 0, 1), vec3(0, 1, 1), vec3(0, 1, 0), vec3(0, 0, 0), vec3(-1, 0, 0)));
		faces.push_back(new Face(vec3(1, 0, 1), vec3(1, 1, 1), vec3(0, 1, 1), vec3(0, 0, 1), vec3(0, 0, 1)));
		faces.push_back(new Face(vec3(0, 0, 0), vec3(0, 1, 0), vec3(1, 1, 0), vec3(1, 0, 0), vec3(0, 0, -1)));
		faces.push_back(new Face(vec3(1, 1, 1), vec3(1, 0, 1), vec3(1, 0, 0), vec3(1, 1, 0), vec3(1, 0, 0)));
		faces.push_back(new Face(vec3(0, 1, 1), vec3(1, 1, 1), vec3(1, 1, 0), vec3(0, 1, 0), vec3(0, 1, 0)));
		
	}

	Hit intersect(const Ray& ray) {
		Hit bestHit;
		for (Intersectable* face : faces) {
			Hit hit = face->intersect(ray); //  hit.t < 0 if no intersection
			if (hit.t > 0 && (bestHit.t < 0 || hit.t < bestHit.t)) { bestHit = hit; }
		}
		if (dot(ray.dir, bestHit.normal) > 0) bestHit.normal = bestHit.normal * (-1);
		return bestHit;
	}
};
Box* box;
struct Cone : Intersectable {
	vec3 n;
	vec3 p;
	float h;
	float cosalpha;

	Cone(vec3 n1, vec3 p1, float h1, float cosf) {
		n = normalize(n1);
		p = p1;
		h = h1;
		cosalpha = cosf;
	}


	Hit intersect(const Ray& ray) {
		Hit hit;

		vec3 s = ray.start;
		vec3 d = ray.dir;
		
		float a = powf(dot(d, n), 2) - dot(d,d)*powf(cosalpha,2);
		float b = 2 * dot(d, n) * dot(s - p, n) - 2 * dot(d, s - p) * powf(cosalpha, 2);
		float c = powf(dot(s - p, n), 2) - dot(s - p, s - p) * powf(cosalpha,2);

		float discr = b * b - 4 * a * c;
		if (discr < 0) { return hit;}
		float t1 = (-b + sqrtf(discr)) / 2 / a;
		float t2 = (-b - sqrtf(discr)) / 2 / a;
		float t = ((t1 < t2 && t1>0)||t2<0) ? t1 : t2;
		if (t <= 0) { 
			return hit;
		}
		if (0 > dot((s + d * t - p), n) || h < dot((s + d * t - p), n))
			t = (t1 > t2) ? t1 : t2;

		if (0 > dot((s + d * t - p), n) || h < dot((s + d * t - p), n)) {
			return hit;
			
		}
		hit.t = t;
		hit.position = s + d * t;
		hit.normal = 2 * dot(hit.position - p, n) * n - 2 * (hit.position - p) * powf(cosalpha, 2);

		return hit;
	}
	void set(vec3 n1, vec3 p1, float h1, float cosf) {
		n = normalize(n1);
		p = p1;
		h = h1;
		cosalpha = cosf;
	}

	void setHit(Hit hit) {
		n = hit.normal;
		p = hit.position;
	}
	vec3 getN() {
		return n;
	}
	vec3 getP() {
		return p;
	}
	float getH() {
		return h;
	}
	float getCosalpha() {
		return cosalpha;
	}
};

Cone* cone1;
Cone* cone2;
Cone* cone3;

class Camera {
	vec3 eye, lookat, right, up;
public:
	void set(vec3 _eye, vec3 _lookat, vec3 vup, float fov) {
		eye = _eye;
		lookat = _lookat;
		vec3 w = eye - lookat;
		float focus = length(w);
		right = normalize(cross(vup, w)) * focus * tanf(fov / 2);
		up = normalize(cross(w, right)) * focus * tanf(fov / 2);
	}
	vec3 geteye() {
		return eye;
	}
	vec3 getLook() {
		return lookat;
	}
	vec3 getVup() {
		return up;
	}
	Ray getRay(int X, int Y) {
		vec3 dir = lookat + right * (2.0f * (X + 0.5f) / windowWidth - 1) + up * (2.0f * (Y + 0.5f) / windowHeight - 1) - eye;
		return Ray(eye, normalize(dir));
	}
};

struct Light {
	vec3 direction;
	vec3 Le;
	Light(vec3 _direction, vec3 _Le) {
		direction = normalize(_direction);
		Le = _Le;
	}
};

float rnd() { return (float)rand() / RAND_MAX; }


const vec3 corrig1 = vec3(2.9f, 1.7, 1);
const vec3 corrig2 = vec3(1.8f, 3.0f,1);
const float ratio = 4;
const float ratio2 = 4;
const float epsilon = 0.0001f;

vec3 La;
static std::vector<Intersectable*> objects;
vec3 eye = vec3(1.8f * 1, 2.0f * 1, 0.75f), vup = vec3(0, 0, 1), lookat = vec3(0.5f, 0.5f, 0.45f);
float fov = 45 * M_PI / 180;
static void pushOcta();
static void pushDoDeca();
static void pushCones();
static void pushBox();
class Scene {
	std::vector<Light*> lights;
	Camera camera;
public:
	void build() {
		camera.set(eye, lookat, vup, fov);
		pushOcta();
		pushDoDeca();
		pushCones();
		pushBox();		
	}
	void setCam(vec3 eye1) {
		camera.set(eye1, camera.getLook(), vec3(0,0,1), 45 * M_PI / 180);
	}
	
	Cone* getClosestCone(int X, int Y) {
		Hit nowHit = firstIntersect(camera.getRay(X, Y));
		Cone* closest = cone1;
		if (length(cone2->getP() - nowHit.position) < length(closest->getP() - nowHit.position)) {
			
			closest = cone2;
		}
		
		if (length(cone3->getP() - nowHit.position) < length(closest->getP() - nowHit.position)) {
			closest = cone3;
		}
		return closest;
	}

	void render(std::vector<vec4>& image) {
		for (int Y = 0; Y < windowHeight; Y++) {
#pragma omp parallel for
			for (int X = 0; X < windowWidth; X++) {
				vec3 color = trace(camera.getRay(X, Y));
				image[Y * windowWidth + X] = vec4(color.x, color.y, color.z, 1);
				nowPrint = false;
			}
		}
	}

	Hit getPointHit(int X, int Y) {
		return firstIntersect(camera.getRay(X, Y));
	}

	Hit secondIntersect(Ray ray) {
		Hit bestHit;
		for (Intersectable* object : objects) {
			Hit hit = object->intersect(ray); //  hit.t < 0 if no intersection
			if (hit.t > 0 && (bestHit.t < 0 || hit.t < bestHit.t))  bestHit = hit;
		}
		if (dot(ray.dir, bestHit.normal) > 0) bestHit.normal = bestHit.normal * (-1);
		return nextIntersect(Ray(ray.start + ray.dir*epsilon + ray.dir * bestHit.t, ray.dir));
	}

	Hit nextIntersect(Ray ray) {
		Hit bestHit;
		for (Intersectable* object : objects) {
			Hit hit = object->intersect(ray); //  hit.t < 0 if no intersection
			if (hit.t > 0 && (bestHit.t < 0 || hit.t < bestHit.t))  bestHit = hit;
		}
		if (dot(ray.dir, bestHit.normal) > 0) bestHit.normal = bestHit.normal * (-1);
		return bestHit;
	}

	Hit firstIntersect(Ray ray) {
		Hit bestHit;
		for (Intersectable* object : objects) {
			Hit hit = object->intersect(ray); //  hit.t < 0 if no intersection
			if (hit.t > 0 && (bestHit.t < 0 || hit.t < bestHit.t)) { bestHit = hit; }
		}
		if (dot(ray.dir, bestHit.normal) > 0) bestHit.normal = bestHit.normal * (-1);
		return bestHit;
	}

	bool shadowIntersect(Ray ray) {	// for directional lights
		for (Intersectable* object : objects) if (object->intersect(ray).t > 0) return true;
		return false;
	}

	bool isListenerShadowed(vec3 from, Cone* cone) {
		for (Intersectable* object : objects) {
			Hit hit = object->intersect(Ray(from, normalize(cone->p+cone->n*.05 - from)));
			if (hit.t > 0 && hit.t < length(cone->p + cone->n * .05 - from))
				return true;
		}
		return false;
	}

	vec3 trace(Ray ray, int depth = 0) {
		Hit hit = firstIntersect(ray);
		vec3 outRadiance = vec3(0, 0, 0);
		if (hit.t < 0) return vec3(0,0,0);

		
		if (!isListenerShadowed(hit.position + hit.normal * 0.008, /*hit.normal,*/ cone1)) {
			float t = (1.73 - length(cone1->p - hit.position)) / 1.73;
			outRadiance = outRadiance + vec3(t, 0, 0);
		}
		if (!isListenerShadowed(hit.position + hit.normal * 0.008, /*hit.normal,*/ cone2)) {
			float t = (1.73 - length(cone2->p - hit.position)) / 1.73;
			outRadiance = outRadiance + vec3(0, t, 0);
		}
		if (!isListenerShadowed(hit.position + hit.normal * 0.008, /*hit.normal,*/ cone3)) {
			float t = (1.73 - length(cone3->p - hit.position)) / 1.73;
			outRadiance = outRadiance + vec3(0, 0, t);
		}
		
		float La2 = 0.2 * (1 + dot(normalize(hit.normal), -1 * normalize(ray.dir)));
		
		
		outRadiance =outRadiance + vec3(La2, La2, La2);
		return outRadiance;
	}
	
};



GPUProgram gpuProgram; // vertex and fragment shaders
Scene scene;

// vertex shader in GLSL
const char* vertexSource = R"(
	#version 330
    precision highp float;

	layout(location = 0) in vec2 cVertexPosition;	// Attrib Array 0
	out vec2 texcoord;

	void main() {
		texcoord = (cVertexPosition + vec2(1, 1))/2;							// -1,1 to 0,1
		gl_Position = vec4(cVertexPosition.x, cVertexPosition.y, 0, 1); 		// transform to clipping space
	}
)";

// fragment shader in GLSL
const char* fragmentSource = R"(
	#version 330
    precision highp float;

	uniform sampler2D textureUnit;
	in  vec2 texcoord;			// interpolated texture coordinates
	out vec4 fragmentColor;		// output that goes to the raster memory as told by glBindFragDataLocation

	void main() {
		fragmentColor = texture(textureUnit, texcoord); 
	}
)";

class FullScreenTexturedQuad {
	unsigned int vao;	// vertex array object id and texture id
	Texture texture;
public:
	FullScreenTexturedQuad(int windowWidth, int windowHeight, std::vector<vec4>& image)
		: texture(windowWidth, windowHeight, image)
	{
		
		glGenVertexArrays(1, &vao);	// create 1 vertex array object
		glBindVertexArray(vao);		// make it active

		unsigned int vbo;		// vertex buffer objects
		glGenBuffers(1, &vbo);	// Generate 1 vertex buffer objects

		// vertex coordinates: vbo0 -> Attrib Array 0 -> vertexPosition of the vertex shader
		glBindBuffer(GL_ARRAY_BUFFER, vbo); // make it active, it is an array
		float vertexCoords[] = { -1, -1,  1, -1,  1, 1,  -1, 1 };	// two triangles forming a quad
		glBufferData(GL_ARRAY_BUFFER, sizeof(vertexCoords), vertexCoords, GL_STATIC_DRAW);	   // copy to that part of the memory which is not modified 
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, NULL);     // stride and offset: it is tightly packed
	}

	void Draw() {
		glBindVertexArray(vao);	// make the vao and its vbos active playing the role of the data source
		gpuProgram.setUniform(texture, "textureUnit");
		glDrawArrays(GL_TRIANGLE_FAN, 0, 4);	// draw two triangles forming a quad
	}
};

FullScreenTexturedQuad* fullScreenTexturedQuad;
void reDraw() {
	std::vector<vec4> image(windowWidth * windowHeight);
	scene.render(image);
	fullScreenTexturedQuad = new FullScreenTexturedQuad(windowWidth, windowHeight, image);
	fullScreenTexturedQuad->Draw();
	glutSwapBuffers();
}
// Initialization, create an OpenGL context
void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);
	scene.build();

	std::vector<vec4> image(windowWidth * windowHeight);
	long timeStart = glutGet(GLUT_ELAPSED_TIME);
	scene.render(image);
	long timeEnd = glutGet(GLUT_ELAPSED_TIME);
	printf("Rendering time: %d milliseconds\n", (timeEnd - timeStart));

	// copy image to GPU as a texture
	fullScreenTexturedQuad = new FullScreenTexturedQuad(windowWidth, windowHeight, image);

	// create program for the GPU
	gpuProgram.create(vertexSource, fragmentSource, "fragmentColor");
}

// Window has become invalid: Redraw
void onDisplay() {
	fullScreenTexturedQuad->Draw();
	glutSwapBuffers();									// exchange the two buffers
}

bool pressed[256] = { false, };
// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) {
	pressed[key] = true;

}

// Key of ASCII code released
void onKeyboardUp(unsigned char key, int pX, int pY) {
	pressed[key] = false;
}

// Mouse click event
void onMouse(int button, int state, int pX, int pY) {
	if (state == 0) {
		Cone* closest = scene.getClosestCone(pX, windowHeight - pY);
		closest->setHit(scene.getPointHit(pX, windowHeight-pY));
		Hit hit = scene.getPointHit(pX, windowHeight - pY);
		reDraw();
	}

	
}

// Move mouse with key pressed
void onMouseMotion(int pX, int pY) {
}

// Idle event indicating that some time elapsed: do animation here -> can move around z axis
void onIdle() {
	if (pressed['a']) {
		float dt = 0.3f;
		eye = vec3((eye.x - lookat.x) * cos(dt) + (eye.y - lookat.y) * sin(dt) + lookat.x,
			-(eye.x - lookat.x) * sin(dt) + (eye.y - lookat.y) * cos(dt) + lookat.y, eye.z);
		scene.setCam(eye);
		reDraw();
	}
	if (pressed['d']) {
		float dt = -0.3f;
		eye = vec3((eye.x - lookat.x) * cos(dt) + (eye.y - lookat.y) * sin(dt) + lookat.x,
			-(eye.x - lookat.x) * sin(dt) + (eye.y - lookat.y) * cos(dt) + lookat.y, eye.z);

		scene.setCam(eye);
		reDraw();
	}

}

static void pushOcta() {
	vec3 dvert1 = (vec3(1, 0, 0) + corrig1) / ratio;
	vec3 dvert2 = (vec3(0, -1, 0) + corrig1) / ratio;
	vec3 dvert3 = (vec3(-1, 0, 0) + corrig1) / ratio;
	vec3 dvert4 = (vec3(0, 1, 0) + corrig1) / ratio;
	vec3 dvert5 = (vec3(0, 0, 1) + corrig1) / ratio;
	vec3 dvert6 = (vec3(0, 0, -1) + corrig1) / ratio;

	Triangle* dface1 = new Triangle(dvert2, dvert1, dvert5);
	Triangle* dface2 = new Triangle(dvert3, dvert2, dvert5);
	Triangle* dface3 = new Triangle(dvert4, dvert3, dvert5);
	Triangle* dface4 = new Triangle(dvert1, dvert4, dvert5);
	Triangle* dface5 = new Triangle(dvert1, dvert2, dvert6);
	Triangle* dface6 = new Triangle(dvert2, dvert3, dvert6);
	Triangle* dface7 = new Triangle(dvert3, dvert4, dvert6);
	Triangle* dface8 = new Triangle(dvert4, dvert1, dvert6);
	objects.push_back(dface1);
	objects.push_back(dface2);
	objects.push_back(dface3);
	objects.push_back(dface4);
	objects.push_back(dface5);
	objects.push_back(dface6);
	objects.push_back(dface7);
	objects.push_back(dface8);
}
static void pushDoDeca() {
	vec3 ddvert1 = (vec3(-.57735, -0.57735, 0.57735) + vec3(corrig2)) / ratio2;
	vec3 ddvert2 = (vec3(0.934172, 0.356822, 0) + vec3(corrig2)) / ratio2;
	vec3 ddvert3 = (vec3(0.93412, -0.356822, 0) + vec3(corrig2)) / ratio2;
	vec3 ddvert4 = (vec3(-0.934172, 0.356822, 0) + vec3(corrig2)) / ratio2;
	vec3 ddvert5 = (vec3(-0.93412, -0.356822, 0) + vec3(corrig2)) / ratio2;
	vec3 ddvert6 = (vec3(0, 0.934172, 0.356822) + vec3(corrig2)) / ratio2;
	vec3 ddvert7 = (vec3(0, 0.93412, -0.356822) + vec3(corrig2)) / ratio2;
	vec3 ddvert8 = (vec3(0.356822, 0, -0.934172) + vec3(corrig2)) / ratio2;
	vec3 ddvert9 = (vec3(-0.356822, 0, -0.934172) + vec3(corrig2)) / ratio2;
	vec3 ddvert10 = (vec3(0, -0.93412, -0.356822) + vec3(corrig2)) / ratio2;
	vec3 ddvert11 = (vec3(0, -0.934172, 0.356822) + vec3(corrig2)) / ratio2;
	vec3 ddvert12 = (vec3(0.356822, 0, 0.934172) + vec3(corrig2)) / ratio2;
	vec3 ddvert13 = (vec3(-0.356822, 0, 0.934172) + vec3(corrig2)) / ratio2;
	vec3 ddvert14 = (vec3(0.57735, 0.5775, -0.57735) + vec3(corrig2)) / ratio2;
	vec3 ddvert15 = (vec3(0.57735, 0.57735, 0.57735) + vec3(corrig2)) / ratio2;
	vec3 ddvert16 = (vec3(-0.57735, 0.5775, -0.57735) + vec3(corrig2)) / ratio2;
	vec3 ddvert17 = (vec3(-0.57735, 0.57735, 0.57735) + vec3(corrig2)) / ratio2;
	vec3 ddvert18 = (vec3(0.5775, -0.5775, -0.57735) + vec3(corrig2)) / ratio2;
	vec3 ddvert19 = (vec3(0.5775, -0.57735, 0.57735) + vec3(corrig2)) / ratio2;
	vec3 ddvert20 = (vec3(-0.5775, -0.5775, -0.57735) + vec3(corrig2)) / ratio2;


	Triangle* ddface1 = new Triangle(ddvert19, ddvert3, ddvert2);
	Triangle* ddface2 = new Triangle(ddvert12, ddvert19, ddvert2);
	Triangle* ddface3 = new Triangle(ddvert15, ddvert12, ddvert2);
	Triangle* ddface4 = new Triangle(ddvert8, ddvert14, ddvert2);
	Triangle* ddface5 = new Triangle(ddvert18, ddvert8, ddvert2);
	Triangle* ddface6 = new Triangle(ddvert3, ddvert18, ddvert2);
	Triangle* ddface7 = new Triangle(ddvert20, ddvert5, ddvert4);
	Triangle* ddface8 = new Triangle(ddvert9, ddvert20, ddvert4);
	Triangle* ddface9 = new Triangle(ddvert16, ddvert9, ddvert4);
	Triangle* ddface10 = new Triangle(ddvert13, ddvert17, ddvert4);
	Triangle* ddface11 = new Triangle(ddvert1, ddvert13, ddvert4);
	Triangle* ddface12 = new Triangle(ddvert5, ddvert1, ddvert4);
	Triangle* ddface13 = new Triangle(ddvert7, ddvert16, ddvert4);
	Triangle* ddface14 = new Triangle(ddvert6, ddvert7, ddvert4);
	Triangle* ddface15 = new Triangle(ddvert17, ddvert6, ddvert4);
	Triangle* ddface16 = new Triangle(ddvert6, ddvert15, ddvert2);
	Triangle* ddface17 = new Triangle(ddvert7, ddvert6, ddvert2);
	Triangle* ddface18 = new Triangle(ddvert14, ddvert7, ddvert2);
	Triangle* ddface19 = new Triangle(ddvert10, ddvert18, ddvert3);
	Triangle* ddface20 = new Triangle(ddvert11, ddvert10, ddvert3);
	Triangle* ddface21 = new Triangle(ddvert19, ddvert11, ddvert3);
	Triangle* ddface22 = new Triangle(ddvert11, ddvert1, ddvert5);
	Triangle* ddface23 = new Triangle(ddvert10, ddvert11, ddvert5);
	Triangle* ddface24 = new Triangle(ddvert20, ddvert10, ddvert5);
	Triangle* ddface25 = new Triangle(ddvert20, ddvert9, ddvert8);
	Triangle* ddface26 = new Triangle(ddvert10, ddvert20, ddvert8);
	Triangle* ddface27 = new Triangle(ddvert18, ddvert10, ddvert8);
	Triangle* ddface28 = new Triangle(ddvert9, ddvert16, ddvert7);
	Triangle* ddface29 = new Triangle(ddvert8, ddvert9, ddvert7);
	Triangle* ddface30 = new Triangle(ddvert14, ddvert8, ddvert7);
	Triangle* ddface31 = new Triangle(ddvert12, ddvert15, ddvert6);
	Triangle* ddface32 = new Triangle(ddvert13, ddvert12, ddvert6);
	Triangle* ddface33 = new Triangle(ddvert17, ddvert13, ddvert6);
	Triangle* ddface34 = new Triangle(ddvert13, ddvert1, ddvert11);
	Triangle* ddface35 = new Triangle(ddvert12, ddvert13, ddvert11);
	Triangle* ddface36 = new Triangle(ddvert19, ddvert12, ddvert11);

	objects.push_back(ddface1);
	objects.push_back(ddface2);
	objects.push_back(ddface3);
	objects.push_back(ddface4);
	objects.push_back(ddface5);
	objects.push_back(ddface6);
	objects.push_back(ddface7);
	objects.push_back(ddface8);
	objects.push_back(ddface9);
	objects.push_back(ddface10);
	objects.push_back(ddface11);
	objects.push_back(ddface12);
	objects.push_back(ddface13);
	objects.push_back(ddface14);
	objects.push_back(ddface15);
	objects.push_back(ddface16);
	objects.push_back(ddface17);
	objects.push_back(ddface18);
	objects.push_back(ddface19);
	objects.push_back(ddface20);
	objects.push_back(ddface21);
	objects.push_back(ddface22);
	objects.push_back(ddface23);
	objects.push_back(ddface24);
	objects.push_back(ddface25);
	objects.push_back(ddface26);
	objects.push_back(ddface27);
	objects.push_back(ddface28);
	objects.push_back(ddface29);
	objects.push_back(ddface30);
	objects.push_back(ddface31);
	objects.push_back(ddface32);
	objects.push_back(ddface33);
	objects.push_back(ddface34);
	objects.push_back(ddface35);
	objects.push_back(ddface36);
}
static void pushCones() {
	cone1 = new Cone(vec3(-0.000000, -0.000000, -1.000000), vec3(0.467208, 0.470877, 1.000000), .1f, .91f);
	cone2 = new Cone(vec3(0.0000, 0.000950, 1.000080), vec3(0.866382, 0.173786, 0.000000), .1f, .91f);
	cone3 = new Cone(vec3(0.850620, 0.000179, 0.525780), vec3(0.640652, 0.745420, 0.319423), .1f, .91f);
	objects.push_back(cone1);
	objects.push_back(cone2);
	objects.push_back(cone3);
}
static void pushBox() {
	box = new Box();
	objects.push_back(box);
}