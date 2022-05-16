struct Vec5
{
	float r = 0.0, u = 0.0, v = 0.0, p = 0.0, T = 0.0;

	__device__ Vec5 operator-(Vec5 other)
	{
		Vec5 res;
		res.r = this->r - other.r;
		res.u = this->u - other.u;
		res.v = this->v - other.v;
		res.p = this->p - other.p;
		res.T = this->T - other.T;
		return res;
	}

	__device__ Vec5 operator+(Vec5 other)
	{
		Vec5 res;
		res.r = this->r + other.r;
		res.u = this->u + other.u;
		res.v = this->v + other.v;
		res.p = this->p + other.p;
		res.T = this->T + other.T;
		return res;
	}

	__device__ Vec5 operator*(float d)
	{
		Vec5 res;
		res.r = this->r * d;
		res.u = this->u * d;
		res.v = this->v * d;
		res.p = this->p * d;
		res.T = this->T * d;
		return res;
	}
};

struct Vec4
{
	float a1 = 0.0, a2 = 0.0, a3 = 0.0, a4 = 0.0;

	__device__ Vec4 operator-(Vec4 other)
	{
		Vec4 res;
		res.a1 = this->a1 - other.a1;
		res.a2 = this->a2 - other.a2;
		res.a3 = this->a3 - other.a3;
		res.a4 = this->a4 - other.a4;
		return res;
	}

	__device__ Vec4 operator+(Vec4 other)
	{
		Vec4 res;
		res.a1 = this->a1 + other.a1;
		res.a2 = this->a2 + other.a2;
		res.a3 = this->a3 + other.a3;
		res.a4 = this->a4 + other.a4;
		return res;
	}

	__device__ Vec4 operator*(float d)
	{
		Vec4 res;
		res.a1 = this->a1 * d;
		res.a2 = this->a2 * d;
		res.a3 = this->a3 * d;
		res.a4 = this->a4 * d;
		return res;
	}

	__device__ Vec4 operator/(float d)
	{
		Vec4 res;
		res.a1 = this->a1 / d;
		res.a2 = this->a2 / d;
		res.a3 = this->a3 / d;
		res.a4 = this->a4 / d;
		return res;
	}
};

struct Vec2
{
	float x = 0.0, y = 0.0;

	__device__ Vec2 operator-(Vec2 other)
	{
		Vec2 res;
		res.x = this->x - other.x;
		res.y = this->y - other.y;
		return res;
	}

	__device__ Vec2 operator+(Vec2 other)
	{
		Vec2 res;
		res.x = this->x + other.x;
		res.y = this->y + other.y;
		return res;
	}

	__device__ Vec2 operator*(float d)
	{
		Vec2 res;
		res.x = this->x * d;
		res.y = this->y * d;
		return res;
	}

	__device__ Vec2 operator/(float d)
	{
		Vec2 res;
		res.x = this->x / d;
		res.y = this->y / d;
		return res;
	}
};

struct Particle{
	Vec2 u;
	float due;
};

struct Color3f
{
	float R = 0.0f;
	float G = 0.0f;
	float B = 0.0f;

	__host__ __device__ Color3f operator+ (Color3f other)
	{
		Color3f res;
		res.R = this->R + other.R;
		res.G = this->G + other.G;
		res.B = this->B + other.B;
		return res;
	}

	__host__ __device__ Color3f operator* (float d)
	{
		Color3f res;
		res.R = this->R * d;
		res.G = this->G * d;
		res.B = this->B * d;
		return res;
	}
};


static Particle* PartFieldOld;
static Particle* PartFieldNew;
static float* PressureFieldOld;
static float* PressureFieldNew;

static Particle* PartFieldHost;
static float* PressureFieldHost;
static uint8_t* dueHost;

__device__ float calcE(float T, float ro, Config Con){
	float dV = Con.dx * Con.dx;
	return Con.Cv*T/ro/dV - ro*Con.a/Con.MolarMass/Con.MolarMass;
}

__device__ float calcT(float ro, float e, Config Con){
	float dV = Con.dx * Con.dx;
	return dV*ro/Con.Cv*(e+ro/Con.a/Con.MolarMass/Con.MolarMass);
	//return Con.T;
}

__device__ float calcP(float ro, float e, Config Con){
	float dV = Con.dx * Con.dx;
	return Con.R*calcT(ro, e, Con)/(dV-Con.b*ro*dV/Con.MolarMass) - Con.a*ro*ro/Con.MolarMass/Con.MolarMass;
	//return Con.P;
}

__device__ int takeIndex(int x, int y){
	return y*FIELDWIDTH + x;
}

__device__ Particle interp(Vec2 v, Particle* PartField){
	float x1 = (int)v.x;
	float y1 = (int)v.y;
	float x2 = (int)v.x + 1;
	float y2 = (int)v.y + 1;

	Particle q1, q2, q3, q4;

	#define CLAMP(val, minv, maxv) min(maxv, max(minv, val))
	#define SET(Q, x, y) Q = PartField[int(CLAMP(y, 0.0f, FIELDHEIGHT - 1.0f)) * FIELDWIDTH + int(CLAMP(x, 0.0f, FIELDWIDTH - 1.0f))]	
	SET(q1, x1, y1);
	SET(q2, x1, y2);
	SET(q3, x2, y1);
	SET(q4, x2, y2);
	#undef SET
	#undef CLAMP
	float t1 = (x2 - v.x) / (x2 - x1);
	float t2 = (v.x - x1) / (x2 - x1);
	Vec2 f1 = q1.u * t1 + q3.u * t2;
	Vec2 f2 = q2.u * t1 + q4.u * t2;

	float C1 = q2.due * t1 + q4.due * t2;
	float C2 = q2.due * t1 + q4.due * t2;
	float t3 = (y2 - v.y) / (y2 - y1);
	float t4 = (v.y - y1) / (y2 - y1);
	Particle res;
	res.u = f1 * t3 + f2 * t4;
	res.due = C1 * t3 + C2 * t4;
	return res;
}

__global__ void advect(Particle* newField, Particle* oldField, float dt, Config Con){
	int x = blockIdx.x * blockDim.x + threadIdx.x+1;
	int y = blockIdx.y * blockDim.y + threadIdx.y+1;
	int index = y*FIELDWIDTH + x;

	float decay = 1.0f / (1.0f + Con.nu * dt);
	Vec2 pos = { x * 1.0f, y * 1.0f };
	Particle& Pold = oldField[index];

	Particle p = interp(pos - Pold.u * dt, oldField);
	p.u = p.u * decay;
	p.due = p.due * decay;
	newField[index] = p;
}


__device__ float JacobiDue(Particle* Field, Vec2 pos, float B, float alpha, float beta){
	float C = Field[int(pos.y) * FIELDWIDTH + int(pos.x)].due;
	float xU = C, xD = C, xL = C, xR = C;
	#define SET(P, x, y) if (x < FIELDWIDTH && x >= 0 && y < FIELDHEIGHT && y >= 0) P = Field[int(y) * FIELDWIDTH + int(x)].due
	SET(xU, pos.x, pos.y - 1);
	SET(xD, pos.x, pos.y + 1);
	SET(xL, pos.x - 1, pos.y);
	SET(xR, pos.x + 1, pos.y);
	#undef SET
	float pressure = (xU + xD + xL + xR + alpha * B) * (1.0f / beta);
	return pressure;
}

__global__ void diffuseDue(Particle* newField, Particle* oldField, float dt, Config Con){
	int x = blockIdx.x * blockDim.x + threadIdx.x+1;
	int y = blockIdx.y * blockDim.y + threadIdx.y+1;
	int index = y*FIELDWIDTH + x;

	Vec2 pos = { x * 1.0f, y * 1.0f };
	float due = oldField[index].due;
	float alpha = Con.DiffCoef * Con.DiffCoef / dt;
	float beta = 4.0f + alpha;
	newField[index].due = JacobiDue(oldField, pos, due, alpha, beta);
}

__device__ Vec2 JacobiVel(Particle* Field, Vec2 v, Vec2 B, float alpha, float beta){
	Vec2 vU = B * -1.0f, vD = B * -1.0f, vR = B * -1.0f, vL = B * -1.0f;
	#define SET(U, x, y) if (x < FIELDWIDTH && x >= 0 && y < FIELDHEIGHT && y >= 0) U = Field[int(y) * FIELDWIDTH + int(x)].u
	SET(vU, v.x, v.y - 1);
	SET(vD, v.x, v.y + 1);
	SET(vL, v.x - 1, v.y);
	SET(vR, v.x + 1, v.y);
	#undef SET
	v = (vU + vD + vL + vR + B * alpha) * (1.0f / beta);
	return v;
}

__global__ void Viscosity(Particle* newField, Particle* oldField, float dt, Config Con){
	int x = blockIdx.x * blockDim.x + threadIdx.x+1;
	int y = blockIdx.y * blockDim.y + threadIdx.y+1;
	int index = y*FIELDWIDTH + x;

	Vec2 pos = { x * 1.0f, y * 1.0f };
	Vec2 u = oldField[index].u;
	float alpha = Con.nu * Con.nu / dt;
	float beta = 4.0f + alpha;
	newField[index].u = JacobiVel(oldField, pos, u, alpha, beta);
}

void Diffusion(dim3 numBlocks, dim3 numThreads, float dt, Config Con){
	for(int i = 0; i<50; i++){
		Viscosity<<<numBlocks, numThreads>>>(PartFieldNew, PartFieldOld, dt, Con);
		diffuseDue<<<numBlocks, numThreads>>>(PartFieldNew, PartFieldOld, dt, Con);

		std::swap(PartFieldNew, PartFieldOld);
	}
}

__device__ float JacobiPress(float* Field, Vec2 pos, float B, float alpha, float beta){
	float C = Field[int(pos.y) * FIELDWIDTH + int(pos.x)];
	float xU = C, xD = C, xL = C, xR = C;
	#define SET(P, x, y) if (x < FIELDWIDTH && x >= 0 && y < FIELDHEIGHT && y >= 0) P = Field[int(y) * FIELDWIDTH + int(x)]
	SET(xU, pos.x, pos.y - 1);
	SET(xD, pos.x, pos.y + 1);
	SET(xL, pos.x - 1, pos.y);
	SET(xR, pos.x + 1, pos.y);
	#undef SET
	float pressure = (xU + xD + xL + xR + alpha * B) * (1.0f / beta);
	return pressure;
}


__device__ float divergency(Particle* field, Vec2 pos){
	float x = pos.x;
	float y = pos.y;
	Particle& C = field[int(y)*FIELDWIDTH+int(x)];
	float x1=-1*C.u.x, x2 = -1*C.u.x, y1 = -1*C.u.y, y2 = -1*C.u.y;
	#define SET(P,x,y) if (x < FIELDWIDTH && x >= 0 && y < FIELDHEIGHT && y >= 0) P = field[int(y) * FIELDWIDTH + int(x)]
	SET(x1,x+1,y).u.x;
	SET(x2,x-1,y).u.x;
	SET(y1,x,y+1).u.y;
	SET(x2,x,y-1).u.y;
	#undef SET
	return (x1-x2+y1-y2)*0.5f;
}


__global__ void CompPressure(Particle* field, float* newField, float* oldField, float dt, Config Con){
	int x = blockIdx.x * blockDim.x + threadIdx.x+1;
	int y = blockIdx.y * blockDim.y + threadIdx.y+1;
	int index = y*FIELDWIDTH + x;

	Vec2 pos = { x * 1.0f, y * 1.0f };
	float div = divergency(field, pos)+0.000001f;
	//printf("%f", div);
	float press = oldField[index];
	float alpha = -1.0;
	float beta = 4.0f;
	newField[index] = JacobiPress(oldField, pos, div, alpha, beta);
}

void computePressure(dim3 numBlocks, dim3 threadsPerBlock, float dt)
{
	for (int i = 0; i < 50; i++)	{
		CompPressure<<<numBlocks, threadsPerBlock>>>(PartFieldOld, PressureFieldNew, PressureFieldOld, dt, config);
		std::swap(PressureFieldOld, PressureFieldNew);
	}
}

__device__ Vec2 gradient(float* field, int x, int y){
	float C = field[y * FIELDWIDTH + x];
	#define SET(P, x, y) if (x < FIELDWIDTH && x >= 0 && y < FIELDHEIGHT && y >= 0) P = field[int(y) * FIELDWIDTH + int(x)]
	float x1 = C, x2 = C, y1 = C, y2 = C;
	SET(x1, x + 1, y);
	SET(x2, x - 1, y);
	SET(y1, x, y + 1);
	SET(y2, x, y - 1);
	#undef SET
	Vec2 res = { (x1 - x2) * 0.5f, (y1 - y2) * 0.5f };
	return res;
}

__global__ void project(Particle* newField, float* pField, float dt)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x+1;
	int y = blockIdx.y * blockDim.y + threadIdx.y+1;
	Vec2& u = newField[y * FIELDWIDTH + x].u;
	u = u - gradient(pField, x, y);
}

__global__ void initField(float StartPress, float StartTemp, float StartRo, Particle* PartField, float* PressureField){
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int index = y*FIELDWIDTH + x;

	if((x<=FIELDHEIGHT) && (y<=FIELDHEIGHT)){
		//if(((x == 0)||(x == FIELDHEIGHT - 1))&&((y == 0)||(y == FIELDWIDTH - 1))){
			PartField[index].u.x = 0;
			PartField[index].u.y = 0;
			PartField[index].due = 0;
			PressureField[index] = 100;
		//}
	}
	if(x==10 && y==10){
		printf("%f \n", PressureField[index]);
	}
}

__global__ void bord_cond(Particle *PartField, float *PressureField, Config Con){
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int index = y*FIELDWIDTH + x;

	if(x == 0 && y < FIELDHEIGHT && y > 0){
		int takeInd = takeIndex(x+1, y);
		PartField[index].u.x = -PartField[takeInd].u.x;
		PartField[index].u.y = PartField[takeInd].u.y;
		PartField[index].due = PartField[takeInd].due;
		PressureField[index] = PressureField[takeInd];
		//PressureField[index] = 0.1;
	}

	if(x == FIELDWIDTH-1 && y < FIELDHEIGHT && y > 0){
		int takeInd = takeIndex(x-1, y);
		//PartField[index].u.x = -PartField[takeInd].u.x;
		PartField[index].u.y = PartField[takeInd].u.y;
		PartField[index].due = PartField[takeInd].due;
		PressureField[index] = PressureField[takeInd];
	}

	if(y == 0 && x < FIELDWIDTH && x > 0){
		int takeInd = takeIndex(x,y+1);
		PartField[index].u.x = PartField[takeInd].u.x;
		PartField[index].u.y = -PartField[takeInd].u.y;
		PartField[index].due = PartField[takeInd].due;
		PressureField[index] = PressureField[takeInd];
	}

	if(y == FIELDHEIGHT-1 && x < FIELDWIDTH && x > 0){
		int takeInd = takeIndex(x,y-1);
		PartField[index].u.x = PartField[takeInd].u.x;
		PartField[index].u.y = -PartField[takeInd].u.y;
		PartField[index].due = PartField[takeInd].due;
		PressureField[index] = PressureField[takeInd];
	}
}

void init_cuda(){
    //Vec5* result;
    int FieldWidth = FIELDWIDTH;
    int FieldHeight = FIELDHEIGHT;

	cudaMalloc((void**)&PartFieldNew, FieldWidth*FieldHeight*sizeof(Particle));
	cudaMalloc((void**)&PartFieldOld, FieldWidth*FieldHeight*sizeof(Particle));
	cudaMalloc((void**)&PressureFieldNew, FieldWidth*FieldHeight*sizeof(Particle));
	cudaMalloc((void**)&PressureFieldOld, FieldWidth*FieldHeight*sizeof(Particle));
	cudaMalloc((void**)&dueHost, 4*FieldWidth*FieldHeight*sizeof(uint8_t));

	//dueHost = (uint8_t*)malloc(4*FIELDHEIGHT*FIELDWIDTH*sizeof(uint8_t));

	dim3 blocks((FIELDWIDTH)/THREDS_PER_BLOCK+1, (FIELDHEIGHT)/THREDS_PER_BLOCK+1);
	dim3 threds(THREDS_PER_BLOCK,THREDS_PER_BLOCK);
	initField<<<blocks, threds>>>(config.P, config.T, config.ro, PartFieldNew, PressureFieldNew);
	initField<<<blocks, threds>>>(config.P, config.T, config.ro, PartFieldOld, PressureFieldOld);
}

__global__ void PaintImage(float *field, Particle* PartField, uint8_t *res){
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int index = y*FIELDWIDTH + x;
	if((x<=FIELDHEIGHT) && (y<=FIELDHEIGHT)){
		res[4 * index+0] = int(PartField[index].u.x/1000);
		//res[4 * index+0] = field[index];
		res[4 * index+1] = 0;
		res[4 * index+2] = 0;
		res[4 * index+3] = 255;
	}

	if(x==100 && y==10){
		//printf("%f \n", PartField[index].u.x);
	}
}

//__global__ void print(float *field, int x, int y){
//	printf("%f \n", field[y*FIELDWIDTH + x]);
//}

__global__ void print(Particle *field, int x, int y){
	printf("%f \n", field[y*FIELDWIDTH + x].u.x);
}

void RenderImage(uint8_t* result, float dt, Vec2 force, Vec2 ForcePoint, float ForceRadius){
	//print<<<1,1>>>(devVec5FieldNew, 0, 1);
	//ForcePoint.x  = 1;
	//ForcePoint.y  = 1;
	print<<<1,1>>>(PartFieldOld, ForcePoint.x, ForcePoint.y);

	dim3 blocks_calc((FIELDWIDTH-2)/THREDS_PER_BLOCK+1, (FIELDHEIGHT-2)/THREDS_PER_BLOCK+1);
	dim3 blocks_rend((FIELDWIDTH)/(THREDS_PER_BLOCK)+1, (FIELDHEIGHT)/(THREDS_PER_BLOCK)+1);
	dim3 threds_calc(THREDS_PER_BLOCK,THREDS_PER_BLOCK);
	dim3 threds_rend(THREDS_PER_BLOCK,THREDS_PER_BLOCK);

	//float* chek;
	//chek = (float*)malloc(FIELDHEIGHT*FIELDWIDTH*sizeof(float));

	bord_cond<<<blocks_rend, threds_rend>>>(PartFieldOld, PressureFieldOld, config);

	Diffusion(blocks_calc, threds_calc, dt, config);

	computePressure(blocks_calc, threds_calc, dt);

	// project
	project<<<blocks_calc, threds_calc>>>(PartFieldOld, PressureFieldOld, dt);
	cudaMemset(PressureFieldOld, 0, FIELDHEIGHT*FIELDWIDTH*sizeof(float));

	// advect
	advect<<<blocks_calc, threds_calc>>>(PartFieldNew, PartFieldOld, dt, config);
	std::swap(PartFieldNew, PartFieldOld);

	PaintImage<<<blocks_rend, threds_rend>>>(PressureFieldOld, PartFieldOld, dueHost);

	cudaMemcpy(result, dueHost, 4*FIELDHEIGHT*FIELDWIDTH*sizeof(uint8_t), cudaMemcpyDeviceToHost);
}

void Exit_cuda(){
	free(PartFieldHost);
	free(PressureFieldHost);

	cudaFree(PartFieldNew);
	cudaFree(PartFieldOld);
	cudaFree(PressureFieldNew);
	cudaFree(PressureFieldOld);
}